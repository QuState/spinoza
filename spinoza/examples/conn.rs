use spinoza::{
    comm::{amp_from_msg, amp_to_msg, DataType, Driver, Node},
    math::Float,
};

const ADDR: &str = "127.0.0.1:34254";

fn main() {
    let server = Driver::new(ADDR);

    std::thread::scope(|s| {
        s.spawn(|| {
            server.run();
        });
        for i in 0..3 {
            s.spawn(move || {
                let mut client = Node::new(ADDR);
                let (buf, data_type) = client.recv();

                match data_type {
                    DataType::Amplitude => {
                        let mut amplitude = amp_from_msg(&buf);
                        println!("client {i} recvd amplitude: {:?}", amplitude);
                        amplitude.re *= i as Float;
                        amplitude.im += i as Float;

                        println!(
                            "client {i} now sending back updated amplitude: {:?}\n",
                            amplitude
                        );
                        client.send(&amp_to_msg(amplitude), DataType::Amplitude);
                    }
                    _ => unimplemented!(),
                }
            });
        }
    });
}
