import queue
import socket
import threading
import tkinter as tk

from display_protocol import DISPLAY_ADDRESS


def main() -> None:
    root = tk.Tk()
    root.title("Live Tracking Display")
    root.geometry("400x300")

    summary_label = tk.Label(
        root,
        text="Waiting for LLM output...",
        justify=tk.LEFT,
        anchor="nw",
        wraplength=360,
    )
    summary_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    updates: queue.Queue[str] = queue.Queue()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(DISPLAY_ADDRESS)
    server.listen()
    server.settimeout(0.5)
    stopping = threading.Event()

    def receive_summaries() -> None:
        while not stopping.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if stopping.is_set():
                    return
                raise

            with connection:
                chunks = []
                while True:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                updates.put(b"".join(chunks).decode("utf-8"))

    def apply_updates() -> None:
        while True:
            try:
                summary = updates.get_nowait()
            except queue.Empty:
                break
            summary_label.config(text=summary)
        root.after(100, apply_updates)

    threading.Thread(target=receive_summaries, daemon=True).start()
    root.after(100, apply_updates)
    try:
        root.mainloop()
    finally:
        stopping.set()
        server.close()


if __name__ == "__main__":
    main()
