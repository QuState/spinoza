use crate::{core::State, math::Float};
use std::{
    collections::HashMap,
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
    Action,
}

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum Action {
    Init,
    ConnectPair,
}

/// Represents a message/instruction sent to a worker node for proper initialization.
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct InitMsg {
    prefix: usize,
    chunk_size: usize,
}

#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct Header {
    msg_size: u64,
    data_type: DataType,
}

unsafe impl bytemuck::Zeroable for Header {}
unsafe impl bytemuck::Pod for Header {}

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

    fn send(&self, tcp_stream: &mut TcpStream, buf: &[u8], data_type: DataType) {
        let num_bytes: u64 = buf.len().try_into().unwrap();
        let header = Header {
            msg_size: num_bytes,
            data_type,
        };

        let header_as_bytes = bytemuck::bytes_of(&header);
        let buf_with_header = [header_as_bytes, buf].concat();
        tcp_stream.write_all(&buf_with_header).unwrap();
    }
}

/// The driver of all distributed quantum computation. All worker nodes carry out
/// instructions sent by the driver.
pub struct Driver {
    sock: TcpListener,
    /// Mapping of prefixes to corresponding connection
    worker_nodes: HashMap<usize, TcpStream>,
}

impl RecvAndSend for Driver {}

impl Driver {
    pub fn new(addr: &str) -> Self {
        let sock = TcpListener::bind(addr).unwrap();
        let worker_nodes = HashMap::new();
        Self { sock, worker_nodes }
    }

    /// Run the driver by accepting connections from worker nodes, and starting
    /// a new thread for each new connection.
    pub fn run(&mut self, n: usize) {
        assert!(n > 0);
        let mut prefix = 0;

        // Create the mapping of prefixes --> connections for all corresponding nodes
        for stream in self.sock.incoming() {
            self.worker_nodes.insert(prefix, stream.unwrap());
            prefix += 1;
        }

        let num_nodes = prefix + 1;
        let chunk_size = (1 << n) / num_nodes;

        // Prep each worker node by assigning each of them a prefix and having each worker node
        // initialize their respective chunk of the Quantum State
        for (prefix, stream) in self.worker_nodes.iter_mut() {
            self.init_worker(*prefix, chunk_size, stream);
        }
    }

    /// Basically, an RPC call that has all worker nodes initiate a zeroed Quantum State. Namely,
    /// each worker node initiates their respective chunk of the, zeroed, Quantum State.
    fn init_worker(&self, prefix: usize, chunk_size: usize, stream: &mut TcpStream) {
        let action = Action::Init;
        let data_type = DataType::Action;
        let init_msg = InitMsg { prefix, chunk_size };

        self.send(stream, buf, DataType::Action);
    }

    fn handle_connection(&self, stream: &mut TcpStream) {
        // self.send();
        let (buf, data_type) = self.recv(stream);

        match data_type {
            _ => unimplemented!(),
        }
    }
}

pub struct Node {
    /// The prefix associated with the node.
    id: Option<usize>,
    driver_node_stream: TcpStream,
    listener: TcpListener,
    state: Option<State>,
}

impl RecvAndSend for Node {}

impl Node {
    /// Connects to the driver node and opens up a listener for other nodes
    pub fn new(driver_addr: &str, listener_addr: &str) -> Self {
        let sock = TcpStream::connect(driver_addr).unwrap();
        let listener = TcpListener::bind(listener_addr).unwrap();

        Self {
            id: None,
            driver_node_stream: sock,
            listener,
            state: None,
        }
    }
}
