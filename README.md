# pyfaas
> Python Function as a Service - custom name pending

We intend to provide a Function as a Service solution that will accomplish the following:
+ run Python functions within ephemeral containers (such as Docker)
+ scaling Python functions across multiple cloud vendors (such as AWS, GCloud, Azure ...)
+ capturing metrics from the running ephemeral containers (for example, throughhttp://opentracing.io/)
+ based on the captured metrics, model the function scalability such as:
  + to automatically create alarms (for scaling) in Amazon CloudWatch
  + scale dynamically with Docker containers


---

### Bibliography
+ https://martinfowler.com/articles/serverless.html
+ https://github.com/cncf/wg-serverless
