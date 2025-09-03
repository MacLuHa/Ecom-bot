from src.Clients.LlmInterface import LlmInterface
from src.configs.LlmConfig import LlmConfig

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.callbacks import get_openai_callback
from typing import Any, Union, List, Dict

import logging
import os
import json
import uuid

class CloudOpenaiClient(LlmInterface):
    
    def __init__(
        self,
        config: LlmConfig,
        chat_log_dir: str = 'logs/',
        faq_file: str = 'data/faq.json',
        orders_file: str = 'data/orders.json',
        system_message: Union[str, None] = None
    ) -> None:
        self._config = config.openai
        self._log_dir = self._is_exist_log_dir(chat_log_dir)
        self._llm = ChatOpenAI(**self._config.model_dump())
        self._llm_with_conversation = None
        self._logger = logging.getLogger(__name__)
        self._system_message = self._get_system_message(system_message)
        self._faq = {d["q"].strip().lower():d["a"] for d in self._get_data_from_json(faq_file)}
        self._orders = self._get_data_from_json(orders_file)
        
    def start_dialog(self) -> None:
        self._model = self._get_model()
        self._chat_loop(self._model)

    def _chat_loop(self, model: Union[ConversationChain, ChatOpenAI]) -> None:
        self._start_session()
        while True:
            try:
                user_text = input("Вы: ")
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                self._logger.error(f"Error: {e}")
                break

            user_text = user_text.strip()
            self._logger.info(f'User: {user_text}')
            if not user_text:
                continue

            cmd = user_text.lower()
            if cmd in ("выход", "стоп", "конец", "exit"):
                print("Bot: До свидания!")
                self._logger.info("User initiated exit. Session ended.")
                break
            if cmd == "сброс":
                if hasattr(model, "memory"):
                    model.memory.clear()
                    self._add_system_message(model)
                print("Bot: Контекст диалога очищен.")
                self._logger.info("User cleared context.")
                continue
            
            if cmd.startswith('/order'):
                order_id = str(cmd.split(' ')[-1])
                self._get_order(order_id)
                continue

            try:
                self._get_answer(cmd)
                
            except Exception as e:
                self._logger.exception("LLM error")
                msg = str(e)
                if "timeout" in msg.lower():
                    print("Бот: [Ошибка] Превышено время ожидания ответа.")
                    self._logger.error(f"Error: {msg}")
                else:
                    print(f"Бот: [Ошибка] {e}")
                    self._logger.error(f"Error: {e}")
                    
                continue


    def _get_model(self) -> Union[ConversationChain, ChatOpenAI]:
        return self._ensure_conversation_chain()

    def _start_session(self) -> None:
        print("Начинаем диалог с ботом (для выхода введите 'выход')")
        id_session = uuid.uuid4()
        print(f"==== New Session {id_session} ====")
        logging.basicConfig(
            filename = os.path.join(self._log_dir, f'session_{id_session}.jsonl'), level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        
    def _ensure_conversation_chain(self) -> ConversationChain:
        if not self._llm_with_conversation:
            self._llm_with_conversation = ConversationChain(
                llm=self._llm,
                memory=ConversationBufferMemory()
            )
            self._add_system_message(self._llm_with_conversation)
        return self._llm_with_conversation

    def _add_system_message(self, model: ConversationChain) -> None:
        model.memory.chat_memory.add_message(SystemMessage(content=self._system_message))

    def _get_answer(self, question: str) -> str:
        if question in self._faq.keys():
            self._logger.info(f'Bot: {self._faq.get(question)}')
            print(f"Bot: {self._faq.get(question)}")
        else:
            with get_openai_callback() as cb:
                reply = self._model.predict(input=question) if isinstance(self._model, ConversationChain) \
                            else self._llm.invoke(question).content
            self._logger.info({'Bot': reply, 'usage': cb.total_tokens})
            print(f'Bot: {reply}')

    def _get_order(self, order_id: int) -> str:
        if order_id in self._orders.keys():
            self._logger.info(f"Bot: Статус заказа {order_id} - {self._orders.get(order_id).get('status')}")
            print(f"Bot: Статус заказа {order_id} - {self._orders.get(order_id).get('status')}")
        else:
            self._logger.info(f"Bot: К сожалению, заказ {order_id} не числится в нашей базе. Пожалуйста, проверьте правильность написания ID заказа")
            print(f"Bot: К сожалению, заказ {order_id} не числится в нашей базе. Пожалуйста, проверьте правильность написания ID заказа")

    def _get_data_from_json(self, path: str) -> Union[List[Dict], Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _get_system_message(self, system_message: Union[str, None]) -> str:
        if system_message and os.path.isfile(system_message):
            with open(system_message, 'r', encoding='utf-8') as f:
                txt = f.read()
            return txt
        else:
            return "Отвечай кратко и по делу. Если не уверен — так и скажи."
        
    def _is_exist_log_dir(self, log_dir: str) -> str:
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
if __name__ == '__main__':
    conf = LlmConfig.load()
    chat = CloudOpenaiClient(conf)
    chat.start_dialog()