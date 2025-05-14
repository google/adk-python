import socket
import subprocess
import platform
import requests
import time
from google.adk.agents import Agent

def check_internet_connection() -> dict:
    """Check internet connection status

    Returns:
        dict: Connection test results
    """
    try:
        # Test DNS resolution
        start_time = time.time()
        socket.gethostbyname("www.google.com")
        dns_time = time.time() - start_time

        # Test HTTP connection
        start_time = time.time()
        response = requests.get("https://www.google.com", timeout=5)
        response_time = time.time() - start_time

        # Check HTTP status code
        if response.status_code != 200:
            return {
                "status": "error",
                "error_message": f"HTTP connection test failed, status code: {response.status_code}"
            }

        # Build report
        report = "Internet connection test:\n"
        report += f"DNS resolution: Success ({dns_time:.2f} seconds)\n"
        report += f"HTTP connection: Success ({response_time:.2f} seconds)\n"
        report += f"Total response time: {(dns_time + response_time):.2f} seconds\n"

        # Evaluate connection quality
        total_time = dns_time + response_time
        if total_time < 0.5:
            quality = "Excellent"
        elif total_time < 1.0:
            quality = "Good"
        elif total_time < 2.0:
            quality = "Average"
        else:
            quality = "Poor"

        report += f"Connection quality: {quality}"

        return {
            "status": "success",
            "report": report
        }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error_message": "HTTP connection timeout, network connection is slow or unstable"
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "error_message": "Unable to establish HTTP connection, may have no internet access"
        }
    except socket.gaierror:
        return {
            "status": "error",
            "error_message": "DNS resolution failed, may have no internet access or DNS server issues"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error checking network connection: {str(e)}"
        }

def ping_host(host: str, count: int = 4) -> dict:
    """Ping specified host to test connection

    Args:
        host (str): Hostname or IP address to ping
        count (int): Number of pings

    Returns:
        dict: Ping test results
    """
    try:
        # Determine OS type
        system = platform.system().lower()

        # Set ping command parameters (based on OS)
        if system == "windows":
            ping_params = f"-n {count}"
        else:  # Linux or macOS
            ping_params = f"-c {count}"

        # Execute ping command
        cmd = f"ping {ping_params} {host}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Process results
        if result.returncode == 0:
            output = result.stdout

            # Try to extract key information (varies by platform)
            response_lines = output.splitlines()
            summary_line = ""

            for line in response_lines:
                if "packets transmitted" in line or "loss" in line:
                    summary_line = line
                    break

            # Build report
            report = f"Ping results to {host}:\n\n"
            report += output

            return {
                "status": "success",
                "report": report
            }
        else:
            return {
                "status": "error",
                "error_message": f"Ping to {host} failed:\n{result.stderr}"
            }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error executing ping: {str(e)}"
        }

def trace_route(host: str) -> dict:
    """Perform route tracing to specified host

    Args:
        host (str): Target hostname or IP address

    Returns:
        dict: Route tracing results
    """
    try:
        # Determine OS type and set command
        system = platform.system().lower()

        if system == "windows":
            cmd = f"tracert {host}"
        else:  # Linux or macOS
            cmd = f"traceroute {host}"

        # Execute command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Process results
        output = result.stdout if result.stdout else result.stderr

        return {
            "status": "success",
            "report": f"Route tracing results to {host}:\n\n{output}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error executing route tracing: {str(e)}"
        }

def get_local_ip_info() -> dict:
    """Get local IP and network interface information

    Returns:
        dict: Local network information
    """
    try:
        # Determine OS type and set command
        system = platform.system().lower()

        if system == "windows":
            cmd = "ipconfig"
        else:  # Linux or macOS
            cmd = "ifconfig"

        # Execute command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Get local hostname
        hostname = socket.gethostname()

        # Try to get public IP
        try:
            public_ip_response = requests.get("https://api.ipify.org", timeout=3)
            public_ip = public_ip_response.text
        except:
            public_ip = "Unable to obtain"

        # Build report
        report = f"Local network information:\n\n"
        report += f"Hostname: {hostname}\n"
        report += f"Public IP: {public_ip}\n\n"
        report += "Network interface details:\n"
        report += result.stdout

        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error getting network information: {str(e)}"
        }

root_agent = Agent(
    name="network_diagnostic_agent",
    model="gemini-2.0-flash",
    description="An intelligent assistant providing network diagnostic and monitoring functions",
    instruction=(
        "You are a network diagnostic assistant that can help users check internet connections, "
        "perform ping tests, trace network routes, and obtain local network information. "
        "Please select appropriate tools based on user needs and help explain the meaning of network test results."
    ),
    tools=[check_internet_connection, ping_host, trace_route, get_local_ip_info]
)
