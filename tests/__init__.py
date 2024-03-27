from langchain.chat_models.fake import FakeListChatModel


class FakeChatOpenAI(FakeListChatModel):
    model_name: str = "model_name_default"
