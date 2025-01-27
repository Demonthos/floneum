#![allow(unused)]
use kalosm::language::*;

// You can derive an efficient parser for your struct with the `Parse` trait
#[derive(Schema, Parse, Clone, Debug)]
struct Account {
    /// A summary of the account holder
    summary: String,
    /// The name of the account holder. This may be the full name of the user or a pseudonym
    name: String,
    /// The age of the account holder
    #[parse(range = 1..=100)]
    age: u8,
}

#[tokio::main]
async fn main() {
    let llm = loop {
        let input = prompt_input("Model (choose from qwen-1.5b, qwen-7b, qwen-14b, llama-8b): ").unwrap();
        match input.as_str() {
            "qwen-1.5b" => break Llama::builder().with_source(LlamaSource::deepseek_r1_distill_qwen_1_5b()).build().await.unwrap(),
            "qwen-7b" => break Llama::builder().with_source(LlamaSource::deepseek_r1_distill_qwen_7b()).build().await.unwrap(),
            "qwen-14b" => break Llama::builder().with_source(LlamaSource::deepseek_r1_distill_qwen_14b()).build().await.unwrap(),
            "llama-8b" => break Llama::builder().with_source(LlamaSource::deepseek_r1_distill_llama_8b()).build().await.unwrap(),
            _ => println!("Invalid model... try again"),
        }
    };

    // Create a task that generates a list of accounts
    let task = llm.task("You generate accounts based on a description of the account holder in this format { \"summary\": \"A summary of the account holder\", \"name\": \"The name of the account holder\", \"age\": numericAgeInYears }");

    // Task can be called like a function with the input to the task. You can await the stream, modify the
    // constraints, or sampler
    let mut response = task("Candice is the CEO of a fortune 500 company. She is a 30 years old.")
        .with_constraints(
            LiteralParser::new("<think>")
                .then(StopOn::new("</think>"))
                .ignore_output_then(Account::new_parser()),
        );

    response.to_std_out().await.unwrap();

    let account = response.await.unwrap();
    println!("{account:#?}");
}
