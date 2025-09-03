from src.Clients.CloudOpenaiClient import CloudOpenaiClient
from src.configs.LlmConfig import LlmConfig

def main():
    config = LlmConfig.load()
    chat = CloudOpenaiClient(config)
    chat.start_dialog()

if __name__ == "__main__":
    main()
