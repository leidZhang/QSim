import logging
from abc import ABC, abstractmethod

from flask import request, jsonify, Response

from system.settings import FAIL_CODE, SUCCESS_CODE
from .service import DataService, IDataService


class IDataHandler(ABC):
    @abstractmethod
    def handle_test_conn(self) -> Response:
        ...

    @abstractmethod
    def handle_step_complete(self) -> Response:
        ...

    @abstractmethod
    def handle_upload_step_data(self) -> Response:
        ...

    @abstractmethod
    def handle_episode_complete(self) -> Response:
        ...


class DataHandler(IDataHandler): # DataController
    def __init__(self) -> None:
        self.service: IDataService = DataService()

    def handle_test_conn(self) -> Response:
        return jsonify({"message": "Hello from server"}), SUCCESS_CODE

    def handle_step_complete(self) -> Response:
        try: 
            binary_data: bytes = request.data
            self.service.handle_step_complete(binary_data)
            return jsonify({"message": "Signal received"}), SUCCESS_CODE            
        except Exception as e:
            logging.error(e)
            return jsonify({"message": f"Error happened in calculate reward due to {e}"}), FAIL_CODE

    def handle_upload_step_data(self) -> Response:
        try:
            binary_data: bytes = request.data
            self.service.handle_upload_step_data(binary_data)
            return jsonify({"message": "Step data received!"}), SUCCESS_CODE
        except Exception as e:
            logging.error(e)
            return jsonify({"message": f"Error happened in receiving step data due to {e}"}), FAIL_CODE

    def handle_episode_complete(self) -> Response:
        try:
            res: str = self.service.handle_episode_complete()
            return jsonify({"message": f"Episode saved at {res}"}), SUCCESS_CODE
        except Exception as e:
            logging.error(e)
            return jsonify({"message": f"Error happened in saving episode data due to {e}"}), FAIL_CODE
        