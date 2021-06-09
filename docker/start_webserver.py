from maui63_postprocessing.web.webserver import UploadPage
import threading

app = UploadPage(__name__)
app.data_file = 'maui63/maui.data'
app.config_file = 'maui63/maui.cfg'
app.weights_file = 'maui63/maui.weights'
app.names_file = 'maui63/maui.names'
app.rvision_url = "<api_url>"

threading.Thread(target=app.run_socketio,
                 kwargs = {'host': '0.0.0.0',
                           'port': 80,
                           'debug': False}
                 ).start()

# TODO: check if git is updated and do that
