import pika 
import os
from flask import Flask, request, jsonify


app = Flask(__name__)
PORT = 5002

QUEUEHOST = os.getenv("QUEUEHOST", "localhost")

app = Flask(__name__)


@app.after_request
def apply_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route('/consume', methods=['GET'])
def consume():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=QUEUEHOST,  # or 'rabbitmq' if using Docker Compose
            credentials=pika.PlainCredentials('guest', 'guest')
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue='predictions')

    method_frame, header_frame, body = channel.basic_get(queue='predictions', 
                                                         auto_ack=True)

    if method_frame:
        message = body.decode()

        try: 
            import ast
            message_json = ast.literal_eval(message)
        except Exception as e: 
            message_json = {"error": "invalid stuff", "raw": message}

        connection.close()
        return jsonify({"message": message_json})
    else:
        connection.close()
        return jsonify({"message": "No messages in queue"}), 204

if __name__ == '__main__':
    print(f"server running on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)
