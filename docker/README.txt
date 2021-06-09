=== ABOUT ===
Docker image files for the Maui63 Webserver

Requirements:
The files below need to be provided to build

data_file : 'maui63/maui.data'
config_file : 'maui63/maui.cfg'
weights_file : 'maui63/maui.weights'
names_file : 'maui63/maui.names'

TODO:
CUDA support (use CUDA/CuDNN base image)

=== BUILD ===
sudo docker build --tag maui_webserver .

==== RUN ====
sudo docker run --name maui_webserver -p 8080:80 maui_webserver