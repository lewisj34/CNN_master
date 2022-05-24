import socket 

def main():
    comp_id = socket.gethostname()
    print(f'Computer ID: {comp_id}')

if __name__ == '__main__':
    main()