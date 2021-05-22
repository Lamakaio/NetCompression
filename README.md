# NetCompression

A small proof of concept of a compile time neural net. 

The jupyter notebook contains the code to train and compress the network (a simple LeNet).

Sparse_embedded contains the neural network library. It is no_std and does not use an allocator, only static arrays.

Embnet contains the code to convert .onnx files to rust code, and the macro to load it.

Longan_net contains a proof of concept network that runs on the longan nano, a small risc board, with no os or allocator. 

Serial_send contains the code to communicate with the longan nano with its serial port. It sends images to the board and receive the board's prediction.
