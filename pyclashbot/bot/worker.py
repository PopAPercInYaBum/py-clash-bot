import traceback
from multiprocessing import Process, Queue
from multiprocessing.synchronize import Event
from typing import Any, Protocol

from pyclashbot.bot.states import StateHistory, StateOrder, state_tree
from pyclashbot.emulators import EmulatorType, get_emulator_registry
from pyclashbot.emulators.base import EmulatorNotReadyError
from pyclashbot.interface.enums import UIField
from pyclashbot.utils.logger import ProcessLogger, attach_worker_file_logging
from pyclashbot.utils.platform import is_macos


class WorkerProcess(Process):
    """Worker process for running the bot.

    Uses multiprocessing instead of threading for reliable force termination.
    Communicates stats back to main process via Queue.
    """

    def __init__(
        self,
        jobs: dict[str, Any],
        stats_queue: Queue,
        shutdown_event: Event,
        session_log_path: str,
    ) -> None:
        super().__init__(daemon=True)
        self.jobs = jobs
        self.stats_queue = stats_queue
        self.shutdown_event = shutdown_event
        self.session_log_path = session_log_path

    def _setup_emulator(self, jobs: dict[str, Any], logger: ProcessLogger):
        """Set up the appropriate emulator based on job configuration."""
        emulator_selection = jobs.get("emulator", EmulatorType.MEMU)
        registry = get_emulator_registry()

        if emulator_selection not in registry:
            print(f"[!] Fatal error: Emulator {emulator_selection} is not supported on this platform!")
            logger.change_status(f"Emulator {emulator_selection} is not available on this platform.")
            return None

        controller_class = registry[emulator_selection]

        try:
            if emulator_selection == EmulatorType.GOOGLE_PLAY:
                logger.change_status("Setting up Google Play device...")
                print("Creating Google Play emulator")
                gp_device_serial = jobs.get(UIField.GP_DEVICE_SERIAL.value) or None
                emu = controller_class(logger=logger, device_serial=gp_device_serial)

            elif emulator_selection == EmulatorType.BLUESTACKS:
                logger.change_status("Setting up BlueStacks 5...")
                print("Creating BlueStacks 5 emulator")
                # Default: Vulkan on macOS, OpenGL on Windows
                default_mode = "vlcn" if is_macos() else "gl"
                bs_mode = jobs.get("bluestacks_render_mode", default_mode)
                render_settings = {"graphics_renderer": bs_mode}
                bs_device_serial = jobs.get(UIField.BS_DEVICE_SERIAL.value) or None
                emu = controller_class(logger=logger, render_settings=render_settings, device_serial=bs_device_serial)

            elif emulator_selection == EmulatorType.MEMU:
                logger.change_status("Checking MEmu installation...")
                print("Creating MEmu emulator")
                from pyclashbot.emulators.memu import verify_memu_installation

                if not verify_memu_installation():
                    logger.change_status("MEmu is not installed — install it to use MEmu Emulator Mode")
                    return None
                render_mode = jobs.get("memu_render_mode", "opengl")
                logger.change_status(f"Setting up MEmu ({render_mode})...")
                emu = controller_class(logger, render_mode)

            elif emulator_selection == EmulatorType.ADB:
                logger.change_status("Connecting to ADB device...")
                print("Creating ADB Device controller")
                adb_serial = jobs.get(UIField.ADB_SERIAL.value) or None
                emu = controller_class(logger=logger, device_serial=adb_serial)

            else:
                return None

            # Construction is cheap; restart() boots the emulator and launches Clash.
            # First-boot failure stops the bot — there is no setup retry.
            logger.change_status("Starting emulator and launching Clash Royale...")
            emu.restart()
            logger.change_status("Emulator ready — starting bot loop")
            return emu

        except EmulatorNotReadyError as e:
            print(f"{emulator_selection} emulator is not ready: {e}")
            logger.change_status(f"{emulator_selection} is not ready — fix it and start the bot again.")
            return None
        except Exception as e:
            print(f"Failed to create {emulator_selection} emulator: {e}")
            logger.change_status(f"Failed to start {emulator_selection} — verify its installation.")
            return None

    def _run_bot_loop(self, emulator, jobs: dict[str, Any], logger: ProcessLogger) -> None:
        """Run the main bot state loop."""
        state = "start"
        state_history = StateHistory(logger)
        state_order = StateOrder()
        consecutive_failures = 0  # only real errors/exceptions count, not clean restarts
        max_consecutive_failures = 5

        while not self.shutdown_event.is_set():
            try:
                new_state = state_tree(emulator, logger, state, jobs, state_history, state_order)

                # A clean restart resets the failure streak — it is a recovery action,
                # not a failure. Only exceptions and "fail" state count toward the limit.
                if new_state == "restart":
                    logger.log(f"Restarting emulator (failure streak: {consecutive_failures})")
                    # Don't increment consecutive_failures here — a restart is intentional.
                elif new_state in ["fail", None]:
                    logger.error(f"Critical error: state_tree returned '{new_state}' — stopping bot")
                    if new_state == "fail":
                        logger.add_restart_after_failure()
                    break
                else:
                    consecutive_failures = 0  # successful state transition — reset

                state = new_state

            except Exception as e:
                consecutive_failures += 1
                tb = traceback.format_exc()
                logger.error(f"Exception in state_tree ({consecutive_failures}/{max_consecutive_failures}): {e}")
                logger.log(f"Current state was: {state}\n{tb}")
                print(f"[ERROR] Exception in state_tree: {e}")
                print(f"[ERROR] Current state was: {state}")
                traceback.print_exc()
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive exceptions — stopping bot to prevent crash loop")
                    break
                # Recover by going back through the restart path
                state = "restart"

            # Note: Pause functionality removed in multiprocessing version
            # If pause is needed, use a separate mp.Event

    def run(self) -> None:
        """Main worker process execution."""
        print("WorkerProcess run()...")
        attach_worker_file_logging(self.session_log_path)

        # Create logger that sends stats through queue
        logger = ProcessLogger(self.stats_queue)

        try:
            emulator = self._setup_emulator(self.jobs, logger)
            if emulator is None:
                return

            self._run_bot_loop(emulator, self.jobs, logger)
        except Exception as err:
            logger.error(str(err))
            traceback.print_exc()
        finally:
            logger.change_status("Bot stopped")


class _StoppableProcess(Protocol):
    """Structural interface for the process-lifecycle methods ``stop_worker_process`` needs.

    Satisfied by ``multiprocessing.Process`` (and ``WorkerProcess``), spawned context
    processes, and test fakes alike — so the helper depends on the contract, not the
    concrete class.
    """

    def is_alive(self) -> bool: ...
    def terminate(self) -> None: ...
    def join(self, timeout: float | None = ...) -> None: ...
    def kill(self) -> None: ...


class _Signal(Protocol):
    """Structural interface for the shutdown event — only ``set()`` is used here."""

    def set(self) -> None: ...


def stop_worker_process(
    process: _StoppableProcess | None,
    shutdown_event: _Signal | None = None,
    *,
    graceful_timeout: float = 2.0,
) -> None:
    """Stop the worker process via OS-level termination.

    Signals shutdown (belt-and-suspenders), then terminates and hard-kills if the
    process does not exit within ``graceful_timeout``. The OS interrupts any blocking
    call in the worker (sleeps, ADB/socket waits), so this does not depend on the
    worker cooperatively noticing the shutdown event.
    """
    if shutdown_event is not None:
        shutdown_event.set()
    if process is None or not process.is_alive():
        return
    process.terminate()  # SIGTERM / TerminateProcess
    process.join(timeout=graceful_timeout)
    if process.is_alive():
        process.kill()  # SIGKILL
        process.join(timeout=graceful_timeout)
