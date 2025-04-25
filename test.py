import socket
import struct
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import threading

# Parameters for the Lorenz system (typical chaotic parameters)
sigma = 10.0
rho = 28.0
beta = 8/3.0

# Time step for Euler's method
dt = 0.01

# Initial conditions (randomized for sender and receiver)
x0_sender, y0_sender, z0_sender = random.random(), random.random(), random.random()
x0_receiver, y0_receiver, z0_receiver = random.random(), random.random(), random.random()

# Lorenz system equations (Euler's method integration)
def lorenz(x, y, z, sigma, rho, beta, dt):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    x_new = x + dx * dt
    y_new = y + dy * dt
    z_new = z + dz * dt
    return x_new, y_new, z_new

# Socket communication setup (using UDP instead of raw socket)
def create_udp_socket(port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Use UDP
        sock.bind(('localhost', port))  # Bind to the given port
        return sock
    except Exception as e:
        print(f"Error creating socket: {e}")
        return None

def send_data(sock, data, dest_ip, dest_port):
    # Pack data into a byte structure for transmission
    packet = struct.pack('!3f', *data)
    sock.sendto(packet, (dest_ip, dest_port))

def receive_data(sock):
    data, addr = sock.recvfrom(1024)
    return struct.unpack('!3f', data)

# Sender: Generates chaotic data and sends to receiver
def sender_program(sock, x, y, z):
    try:
        send_data(sock, (x, y, z), 'localhost', 12346)
        for _ in range(100):
            x, y, z = lorenz(x, y, z, sigma, rho, beta, dt)
            send_data(sock, (x, y, z), 'localhost', 12346)
            time.sleep(0.1)
    finally:
        sock.close()  # Ensure socket is closed
    return x, y, z

def receiver_program(sock, x, y, z):
    try:
        for _ in range(100):
            data = receive_data(sock)
            print(f"Received state: {data}")
            x, y, z = lorenz(x, y, z, sigma, rho, beta, dt)
            print(f"Receiver state: ({x}, {y}, {z})")
            error = abs(data[0] - x) + abs(data[1] - y) + abs(data[2] - z)
            if error > 0.5:
                print(f"Error detected: {error}. Attempting resynchronization.")
                x, y, z = data
            time.sleep(0.1)
    finally:
        sock.close()  # Ensure socket is closed
    return x, y, z


# Plotting the Lorenz attractor with real-time updates
def plot_trajectory(sender_trajectory, receiver_trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Real-Time Chaotic Synchronization')

    # Initialize plot lines for sender and receiver
    sender_line, = ax.plot([], [], [], 'b', label='Sender')
    receiver_line, = ax.plot([], [], [], 'r', label='Receiver')
    
    ax.legend()
    
    # Update function for animation
    def update_plot(frame):
        sender_trajectory.append(frame[0])  # Update sender trajectory
        receiver_trajectory.append(frame[1])  # Update receiver trajectory
        
        sender_x, sender_y, sender_z = zip(*sender_trajectory)
        receiver_x, receiver_y, receiver_z = zip(*receiver_trajectory)
        
        sender_line.set_data(sender_x, sender_y)
        sender_line.set_3d_properties(sender_z)

        receiver_line.set_data(receiver_x, receiver_y)
        receiver_line.set_3d_properties(receiver_z)

        return sender_line, receiver_line
    
    ani = animation.FuncAnimation(fig, update_plot, frames=zip(sender_trajectory, receiver_trajectory),
                                  interval=100, blit=False)
    plt.show()

if __name__ == "__main__":
    choice = input("Enter 'sender' to start the sender program or 'receiver' to start the receiver program: ")
    
    if choice == 'sender':
        sender_sock = create_udp_socket(12344)  # Sender uses port 12345
        receiver_sock = create_udp_socket(12343)  # Receiver uses port 12346
        
        sender_trajectory = [(x0_sender, y0_sender, z0_sender)]
        receiver_trajectory = [(x0_receiver, y0_receiver, z0_receiver)]
        
        sender_thread = threading.Thread(target=sender_program, args=(sender_sock, x0_sender, y0_sender, z0_sender))
        receiver_thread = threading.Thread(target=receiver_program, args=(receiver_sock, x0_receiver, y0_receiver, z0_receiver))
        
        sender_thread.start()
        receiver_thread.start()

        sender_thread.join()
        receiver_thread.join()

        plot_trajectory(sender_trajectory, receiver_trajectory)

    elif choice == 'receiver':
        receiver_sock = create_udp_socket(12346)  # Receiver uses port 12346
        sender_sock = create_udp_socket(12345)  # Sender uses port 12345
        
        sender_trajectory = [(x0_sender, y0_sender, z0_sender)]
        receiver_trajectory = [(x0_receiver, y0_receiver, z0_receiver)]
        
        receiver_thread = threading.Thread(target=receiver_program, args=(receiver_sock, x0_receiver, y0_receiver, z0_receiver))
        sender_thread = threading.Thread(target=sender_program, args=(sender_sock, x0_sender, y0_sender, z0_sender))
        
        receiver_thread.start()
        sender_thread.start()

        receiver_thread.join()
        sender_thread.join()

        plot_trajectory(sender_trajectory, receiver_trajectory)

    else:
        print("Invalid choice.")
