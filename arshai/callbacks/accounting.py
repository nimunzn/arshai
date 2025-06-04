from arshai.clients.arshai.accounting import accounting_client

class AccountingCallbackHandler():
    def __init__(self, correlation_id=None, request_id=None, user_id=None) -> None:
        super().__init__()
        self._correlation_id = correlation_id
        self._request_id = request_id
        self._user_id = user_id
        self._prompt_txt = ''
        self._output_txt = ''
        self.model_name = None
        print("Custom Accountig handler Initialized")

    async def call_accounting(self, output_tokens, prompt_tokens, agent_slug):
        accounting_client.usage_log(
        request_id=self._request_id,
        correlation_id=self._correlation_id,
        incoming_used_tokens=prompt_tokens,
        outgoing_used_tokens=output_tokens,
        # used_seconds=None,
        agent_slug=agent_slug,
        user_id=self._user_id,
        )
