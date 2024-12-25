terraform {
  backend "s3" {
    bucket = "interactive_yuma_simulator-bdgvhz"
    key    = "prod/main.tfstate"
    region = "us-east-1"
  }
}
