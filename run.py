from app import app
from waitress import serve


if __name__ == "__main__":

    print("=" * 50)
    print(" Student Performance Prediction App")
    print("=" * 50)
    print(" Running on http://127.0.0.1:5001")
    print(" Press CTRL+C to quit")
    print("=" * 50)

    serve(
        app,
        host="127.0.0.1",
        port=5001
    )