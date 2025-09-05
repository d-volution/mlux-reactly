from assistant import Agent_Chat, Agent_ReAct, ResponseStream
import sys

# setup code:

class ChatResponseStream(ResponseStream):
    def __init__(self):
        self.written_flag: bool = False

    def write(self, text_chunk):
        print(text_chunk, end="", flush=True)
        self.written_flag = True


response_stream = ChatResponseStream()

if len(sys.argv) > 1 and sys.argv[1] == "chat":
    agent = Agent_Chat(response_stream=response_stream)
else:
    agent = Agent_ReAct(response_stream=response_stream)




# chat loop:

while True:
    user_input = input(">>> ")

    if user_input.startswith("/bye"):
        break

    response = agent.query(user_input)

    if response_stream.written_flag:
        print()
        response_stream.written_flag = False
    print(response)

