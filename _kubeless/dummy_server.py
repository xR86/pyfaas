#!/usr/bin/env python3
import cherrypy
import os
import json

class HelloWorld(object):
    @cherrypy.expose
    def index(self):
        resp = os.getenv("RESP", None)
        if resp != None:
            return json.dumps({"predicted": int(resp)})
        return open("resp").read()

cherrypy.server.socket_host = '0.0.0.0'
cherrypy.quickstart(HelloWorld())
