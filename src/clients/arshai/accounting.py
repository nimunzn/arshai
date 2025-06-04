import logging
import os
import requests
logger = logging.getLogger(__name__)


class AccountingClient:
    _accounting_base_url = os.getenv("ACCOUNTING_URL", "http://localhost:20101/")
    _usage_log_path = "api/v1/core/generic/agent-usage-transaction"
    _rate_limit_path = "api/v1/core/generic/realm-is-rate-limited"

    def __init__(self):
        self.client = requests.Session()
        self.client.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def _make_request(self, endpoint, method='GET', data=None, params=None,headers=None):
        url = f"{self._accounting_base_url}{endpoint}"
        try:
            # Merge additional headers if provided
            final_headers = self.client.headers.copy()
            if headers:
                final_headers.update(headers)
            response = self.client.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=final_headers,
            )
            logger.error(f"*** json response from accounting: {response.json()} ***")
            response.raise_for_status()
            return response
        except Exception as e:
            # logger.error(f"*** error in accounting: {e.message} ***")
            raise e

    def usage_log(
            self,
            request_id: str,
            correlation_id: str,
            incoming_used_tokens: str,
            outgoing_used_tokens: str,
            # used_seconds: str,
            agent_slug: str,
            user_id: str,
    ):
        data = {
            "request_id": request_id,
            "correlation_id": correlation_id,
            "incoming_used_tokens": incoming_used_tokens,
            "outgoing_used_tokens": outgoing_used_tokens,
            # "used_seconds": used_seconds,
            "agent_slug": agent_slug,
            "user_id": user_id,
        }
        # Add X-Correlation header dynamically
        headers = {
            "X-Correlation": correlation_id,
            "X-Correlation-Id": correlation_id  # Add the correlation ID to the headers
            # "X-Auth-User": json.dumps({"user_id": user_id}) 
        }
        logger.info(f"data is: {data}")
        res = self._make_request(endpoint=self._usage_log_path, method='POST', data=data, headers=headers)
        return res.status_code, res.json()

accounting_client = AccountingClient()
