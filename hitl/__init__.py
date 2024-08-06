from .wrappers import KbdWrapper

def run_hitl(done) -> None:
    try:
        wrapper: KbdWrapper = KbdWrapper()
        while not done.is_set():
            wrapper.execute()
        wrapper.terminate()
    except KeyboardInterrupt:
        wrapper.terminate()
        print("HITL terminated")