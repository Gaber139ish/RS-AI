import asyncio
import json
from typing import List, Tuple


class NodeNet:
    def __init__(self, listen_host: str = '127.0.0.1', listen_port: int = 7000, peers: List[Tuple[str, int]] | None = None):
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        self.peers = peers or []
        self._server = None
        self._tasks: List[asyncio.Task] = []
        self._connections: List[asyncio.StreamWriter] = []

    async def start(self):
        self._server = await asyncio.start_server(self._handle_conn, host=self.listen_host, port=self.listen_port)
        for host, port in self.peers:
            self._tasks.append(asyncio.create_task(self._dial(host, port)))

    async def stop(self):
        for w in list(self._connections):
            try:
                w.close()
                await w.wait_closed()
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for t in self._tasks:
            t.cancel()

    async def _dial(self, host: str, port: int):
        try:
            r, w = await asyncio.open_connection(host, port)
            self._connections.append(w)
        except Exception:
            await asyncio.sleep(0.5)

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._connections.append(writer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode('utf-8'))
                    # Placeholder: in a real system we would route messages to handlers
                except Exception:
                    continue
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            if writer in self._connections:
                self._connections.remove(writer)

    async def broadcast(self, payload: dict):
        data = (json.dumps(payload) + "\n").encode('utf-8')
        dead = []
        for w in self._connections:
            try:
                w.write(data)
                await w.drain()
            except Exception:
                dead.append(w)
        for w in dead:
            if w in self._connections:
                self._connections.remove(w)