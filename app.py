"""결과 뷰어 실행: python app.py [--outputs outputs] [--data data] [--port 8000] [--no-browser]

실제 코드는 viewer/ 에 있다 (exe 로 묶을 때의 진입점이기도 하다).
"""
from viewer.server import main

if __name__ == "__main__":
    main()
