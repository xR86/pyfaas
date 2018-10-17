# pyfaas
> Python Function as a Service - custom name pending

We intend to provide a Function as a Service solution that will accomplish the following:
+ run Python functions within ephemeral containers (such as Docker)
+ scaling Python functions across multiple cloud vendors (such as AWS, GCloud, Azure ...)
+ capturing metrics from the running ephemeral containers (for example, through     http://opentracing.io/)
+ based on the captured metrics, model the function scalability such as:
  + to automatically create alarms (for scaling) in Amazon CloudWatch
  + scale dynamically with Docker containers

## Useful links
+ current LaTeX documentation available both in [docs/paper.tex](docs/paper.tex), and on [Overleaf](https://www.overleaf.com/read/zbsqyvbmfxqw) (read-only)


---

### Bibliography

Articles:
+ https://martinfowler.com/articles/serverless.html
+ https://github.com/cncf/wg-serverless

Solutions:
+ https://github.com/serverless/serverless - https://serverless.com/
+ https://github.com/iron-io/functions
