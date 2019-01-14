#!/usr/bin/env python3
import cherrypy
import os
import json

class HelloWorld(object):

    DATA = 10

    def get_resp(self):
        resp = os.getenv("RESP", None)
        if self.DATA is None:
            return resp
        return self.DATA

    @cherrypy.expose
    def index(self):
        resp = self.get_resp()
        if resp != None:
            return json.dumps({"predicted": int(resp)})
        return open("resp").read()

    @cherrypy.expose
    def set(self, data):
        try:
            if data is not None:
                self.DATA = int(data)
        except Exception:
            self.DATA = None
        return "OK"
                


cherrypy.server.socket_host = '0.0.0.0'
cherrypy.quickstart(HelloWorld())
