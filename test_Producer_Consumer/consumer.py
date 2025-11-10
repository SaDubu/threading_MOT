class Consumer:
    def __init__(self, q):
        self.q = q

    def run(self):
        while True:
            item = self.q.get()  # 블로킹 대기
            if item is None:
                print("[Consumer] 🔚 종료 신호 수신")
                break
            print(f"[Consumer] ⬅️ 큐에서 {item} 꺼냄")
        print("[Consumer] ✅ 작업 완료")
