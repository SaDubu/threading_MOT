import time

class Producer:
    def __init__(self, q):
        self.q = q

    def run(self):
        for i in range(5):
            print(f"[Producer] ➡️ 큐에 {i} 넣음")
            self.q.put(i)
            time.sleep(1)
        self.q.put(None)  # 종료 신호
        print("[Producer] ✅ 작업 완료")
