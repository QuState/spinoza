use crate::math::Float;
use std::{
    io::prelude::*,
    net::{TcpListener, TcpStream},
};

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct Amplitude {
    pub re: Float,
    pub im: Float,
}

unsafe impl bytemuck::Zeroable for Amplitude {}
unsafe impl bytemuck::Pod for Amplitude {}

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum DataType {
    State,
    Side0,
    Side1,
    Amplitude,
}

#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct Header {
    msg_size: u64,
    data_type: DataType,
}

unsafe impl bytemuck::Zeroable for Header {}
unsafe impl bytemuck::Pod for Header {}

pub struct Driver {
    sock: TcpListener,
}

trait RecvAndSend {
    fn recv_header(&self, tcp_stream: &mut TcpStream) -> Header {
        const n: usize = std::mem::size_of::<Header>();
        let mut data = [0u8; n];
        let mut total_recvd = 0;

        while total_recvd < n {
            total_recvd += tcp_stream.read(&mut data).unwrap();
        }

        *bytemuck::from_bytes::<Header>(&data)
    }

    /// Helper function to recv `n` number of bytes into a Vector of bytes
    fn recv_n(&self, tcp_stream: &mut TcpStream, n: u64) -> Vec<u8> {
        let n: usize = n.try_into().unwrap();
        let mut data = vec![0u8; n];
        let mut total_recvd = 0;

        while total_recvd < n {
            total_recvd += tcp_stream.read(&mut data).unwrap();
        }
        data
    }

    /// Receive the message size, n, and receive n bytes into a buffer
    fn recv(&self, tcp_stream: &mut TcpStream) -> (Vec<u8>, DataType) {
        let header = self.recv_header(tcp_stream);
        (self.recv_n(tcp_stream, header.msg_size), header.data_type)
    }

    fn send(&self, tcp_stream: &mut TcpStream, buf: &[u8], data_type: DataType) -> usize {
        let num_bytes: u64 = buf.len().try_into().unwrap();
        let header = Header {
            msg_size: num_bytes,
            data_type,
        };

        let header_as_bytes = bytemuck::bytes_of(&header);
        let buf_with_header = [header_as_bytes, buf].concat();
        tcp_stream.write(&buf_with_header).unwrap()
    }
}

impl RecvAndSend for Driver {}

impl Driver {
    pub fn new(addr: &str) -> Self {
        let sock = TcpListener::bind(addr).unwrap();
        Self { sock }
    }

    pub fn run(&self) {
        std::thread::scope(|s| {
            for stream in self.sock.incoming() {
                s.spawn(|| {
                    let mut stream = stream.unwrap();
                    self.handle_connection(&mut stream)
                });
            }
        });
    }

    fn handle_connection(&self, stream: &mut TcpStream) {
        let amplitude: &[u8] = &amp_to_msg(Amplitude { re: 1.0, im: 0.0 });
        self.send(stream, amplitude, DataType::Amplitude);

        let (buf, data_type) = self.recv(stream);

        match data_type {
            DataType::Amplitude => {
                let amplitude = amp_from_msg(&buf);
                println!("server recvd amplitude: {:?}\n", amplitude);
            }
            _ => unimplemented!(),
        }
    }
}

/// For POC only
pub fn amp_to_msg(amp: Amplitude) -> Vec<u8> {
    let amp_as_bytes = bytemuck::bytes_of(&amp);
    amp_as_bytes.to_vec()
}

/// For POC only
pub fn amp_from_msg(buf: &[u8]) -> Amplitude {
    *bytemuck::from_bytes::<Amplitude>(buf)
}

pub struct Node {
    sock: TcpStream,
}

impl RecvAndSend for Node {}

impl Node {
    pub fn new(addr: &str) -> Self {
        let sock = TcpStream::connect(addr).unwrap();
        Self { sock }
    }

    fn recv_header(&mut self) -> Header {
        const n: usize = std::mem::size_of::<Header>();
        let mut data = [0u8; n];
        let mut total_recvd = 0;

        while total_recvd < n {
            total_recvd += self.sock.read(&mut data).unwrap();
        }

        *bytemuck::from_bytes::<Header>(&data)
    }

    /// Helper function to recv `n` number of bytes into a Vector of bytes
    fn recv_n(&mut self, n: u64) -> Vec<u8> {
        let n: usize = n.try_into().unwrap();
        let mut data = vec![0u8; n];
        let mut total_recvd = 0;

        while total_recvd < n {
            total_recvd += self.sock.read(&mut data).unwrap();
        }
        data
    }

    /// Receive the message size, n, and receive n bytes into a buffer
    pub fn recv(&mut self) -> (Vec<u8>, DataType) {
        let header = self.recv_header();
        (self.recv_n(header.msg_size), header.data_type)
    }

    pub fn send(&mut self, buf: &[u8], data_type: DataType) -> usize {
        let num_bytes: u64 = buf.len().try_into().unwrap();
        let header = Header {
            msg_size: num_bytes,
            data_type,
        };

        let header_as_bytes = bytemuck::bytes_of(&header);
        let buf_with_header = [header_as_bytes, buf].concat();
        self.sock.write(&buf_with_header).unwrap()
    }
}
