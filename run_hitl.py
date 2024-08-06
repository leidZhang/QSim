from threading import Event

from hitl import run_hitl


if __name__ == "__main__":
    done: Event = Event()
    run_hitl(done)