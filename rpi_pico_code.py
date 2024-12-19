import network
import socket
import time
from machine import Pin
import onewire, ds18x20

# Wi-Fi credentials
SSID = 'DemoUser'   # Fill with your own
PASSWORD = '12345678' # Fill with your own

# Connect to Wi-Fi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)

print("Connecting to Wi-Fi...")
while not wlan.isconnected():
    time.sleep(1)

print("Wi-Fi connected:", wlan.ifconfig())

# DS18B20 setup
ds_pin = Pin(21)  # GPIO pin connected to the DS18B20 data line
ds_sensor = ds18x20.DS18X20(onewire.OneWire(ds_pin))

roms = ds_sensor.scan()
if not roms:
    print("No DS18B20 sensor found!")
    exit()
else:
    print("DS18B20 sensor detected:", roms)

# Server details
SERVER_IP = '192.168.101.88'  # Replace with your PC's IP address
SERVER_PORT = 12345           # Same port as the server

# Create a socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER_IP, SERVER_PORT))
print(f"Connected to server at {SERVER_IP}:{SERVER_PORT}")

# Initial temperature value for comparison
temp = None

try:
    while True:
        ds_sensor.convert_temp()  # Start temperature conversion

        for rom in roms:
            temperature = ds_sensor.read_temp(rom)
            # message = f"Temperature: {temperature:.2f}°C"

            # Determine breathing status
            if temp is not None:
                if temperature - temp >=0.4:
                    message = "Exhale"
                elif temp - temperature >= 0.35:
                    message = "Inhale"
            # Print and send message
            print(message)
            client.sendall(message.encode())

            # Update the temp variable for the next comparison
            temp = temperature

            # Wait for server response
            response = client.recv(1024)
            print("Server response:", response.decode())

except KeyboardInterrupt:
    print("\nClient shutting down")
finally:
    client.close()

