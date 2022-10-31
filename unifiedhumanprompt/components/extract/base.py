class Extract(object):

    @staticmethod
    def extract(raw_response, **kwargs):
        """
        Extract the answer from the raw response.
        Args:
            raw_response: raw response from model
            **kwargs: other arguments
        Returns: extracted result
        """
        return raw_response.strip()
