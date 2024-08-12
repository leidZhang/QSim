from restful import IDataHandler, DataHandler
# from .__init__ import app


def setup_routes(app):
    handler: IDataHandler = DataHandler() # controller in MVC architecture
    app.add_url_rule("/", view_func=handler.handle_test_conn, methods=["GET"])
    app.add_url_rule("/step_complete", view_func=handler.handle_step_complete, methods=["POST"])
    app.add_url_rule("/upload_step_data", view_func=handler.handle_upload_step_data, methods=["POST"])
    app.add_url_rule("/episode_complete", view_func=handler.handle_episode_complete, methods=["POST"])
