import threading
import queue
from producer import Producer
from consumer import Consumer

def main():
    q = queue.Queue()  # thread-safe

    producer = Producer(q)
    consumer = Consumer(q)

    # 스레드 생성
    producer_thread = threading.Thread(target=producer.run)
    consumer_thread = threading.Thread(target=consumer.run)

    # 스레드 시작
    producer_thread.start()
    consumer_thread.start()

    # 종료 대기
    producer_thread.join()
    consumer_thread.join()

    print("🎉 모든 작업 종료")

if __name__ == "__main__":
    main()
