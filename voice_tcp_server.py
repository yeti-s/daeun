import os
import socket
import logging
import argparse
from utils import send, receive
from voice.tts import TTSModel

# create logger
os.makedirs('log', exist_ok=True)
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
file_handler = logging.FileHandler(os.path.join('log', 'voice.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def listen(host, port, model):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
        s_sock.bind((host, port))
        s_sock.listen()
        logger.info(f'Listening on {host}:{port}')

        c_sock, addr = s_sock.accept()
        with c_sock:
            logger.info(f'Connected by {addr}')
            while True:
                data = receive(c_sock)
                if not data:
                    logger.info(f'Close connection with {addr}')
                    return True
                try:
                    utterance = data.decode('utf-8')
                    logger.info(f'Receive: {utterance}')
                    
                    audio_bin = model.generate(utterance)
                    send(c_sock, audio_bin)
                    logger.info(f'Send {len(audio_bin)} bytes to {addr}')
                except Exception as e:
                    logger.error(str(e))

def main(args):
    model = TTSModel(args.model, args.config, args.speed, args.device)
    logger.info(f'Initiate TTS model. model:{args.model}, config:{args.config}, device:{args.device}')
    listen(args.host, args.port, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default="0.0.0.0", help="host")
    parser.add_argument('--port', type=int, default=9999, help="port")
    parser.add_argument('--speed', type=float, default=1., help="sound playback speed")
    parser.add_argument('--device', default='cuda:0', help="inference device")
    parser.add_argument('--model', required=True, help="model path")
    parser.add_argument('--config', required=True, help="model config")
    main(parser.parse_args())