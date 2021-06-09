//Client-side Javascript code for handling random numbers
$(document).ready(function(){
    //connect to the socket server.
    //var socket = io('/stdout_log');
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/stdout_log')
    var strings_received = [];
    
    socket.on('connect', function() {
        socket.emit('my_event', {data: 'connected to the SocketServer...'});
    });

    //receive details from server
    socket.on('process_stdout', function(msg) {
        console.log("Got stdout: " + msg.string);
        //maintain a list of ten numbers
        if (strings_received.length >= 10){
            strings_received.shift()
        }
        strings_received.push(msg.string);
        stdout_string = '';
        for (var i = 0; i < strings_received.length; i++){
            stdout_string = stdout_string + '<p>' + strings_received[i].toString() + '</p>';
        }
        $('#stdout_log').html(stdout_string);
    });

});