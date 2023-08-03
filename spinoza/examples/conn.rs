use spinoza::{
    comm::{amp_from_msg, amp_to_msg, DataType, Driver, Node},
    math::Float,
};

const DRIVER_ADDR: &str = "127.0.0.1:34254";
const NODE0_ADDR: &str = "127.0.0.1:34255";
const NODE1_ADDR: &str = "127.0.0.1:34256";
const NODE2_ADDR: &str = "127.0.0.1:34257";
const NODE3_ADDR: &str = "127.0.0.1:34258";
const NODE4_ADDR: &str = "127.0.0.1:34259";

fn main() {
    let driver = Driver::new(DRIVER_ADDR);

    let mut client0 = Node::new(DRIVER_ADDR, NODE0_ADDR);
    let mut client1 = Node::new(DRIVER_ADDR, NODE1_ADDR);
    let mut client2 = Node::new(DRIVER_ADDR, NODE2_ADDR);
    let mut client3 = Node::new(DRIVER_ADDR, NODE3_ADDR);

    std::thread::scope(|s| {
        s.spawn(|| {
            driver.run();
        });
    });
}
