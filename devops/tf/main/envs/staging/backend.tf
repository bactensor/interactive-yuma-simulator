terraform {
  backend "s3" {
    bucket = "interactive_yuma_simulator-bdgvhz"
    key    = "staging/main.tfstate"
    region = "us-east-1"
  }
}
