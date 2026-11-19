import os

from dashboard.application.main_app import app
import dashboard.application.main_callback  # ez a sor kell ahhoz, hogy működjenek a callback-ek


if __name__ == "__main__":
    app.run(
        host=os.getenv("DASH_HOST", "127.0.0.1"),
        port=int(os.getenv("DASH_PORT", "8050")),
        debug=os.getenv("DASH_DEBUG", "0") == "1",
    )
