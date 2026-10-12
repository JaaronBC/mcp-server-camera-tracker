
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class GPSHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/location":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length))

        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        altitude_data = data.get("altitude")
        altitude = float(altitude_data) if altitude_data is not None else None

        if not (-90 <= latitude <= 90):
            self.send_error(400, "Invalid latitude")
            return

        if not (-180 <= longitude <= 180):
            self.send_error(400, "Invalid longitude")
            return

        location = {
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
        }

        with open("camera_location.json", "w") as file:
            json.dump(location, file, indent=2)

        print("Received GPS location:", location)

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Location received")

    def log_message(self, format, *args):
        pass


server = HTTPServer(("0.0.0.0", 8000), GPSHandler)
print("GPS receiver listening on port 8000")
server.serve_forever()
