"""Dispatcher security contract: no live GitHub calls."""
import unittest
from unittest.mock import patch
from integrations.github_dispatcher import server

class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.api=lambda method,path,token,payload=None: self.fake(method,path,token,payload)

    def fake(self,method,path,token,payload):
        self.calls.append((method,path,payload))
        return {"total_count":0} if method=="GET" else None

    def test_fail_closed_default(self):
        with self.assertRaisesRegex(ValueError,"allowlisted"):
            server.dispatch("dev4-reusable-selection.yml","main",{"round":"3"},
                            token="dummy",requester_approved=True,api=self.api)
        self.assertEqual(self.calls,[])

    def test_explicit_authorization_and_exact_inputs(self):
        with patch.dict(server.ALLOWED,{"dev4-reusable-selection.yml":{"round":3}}):
            for approved,workflow,ref,inputs in [
                (False,"dev4-reusable-selection.yml","main",{"round":"3"}),
                (True,"dev4-round2-qant-cpu.yml","main",{"round":"3"}),
                (True,"dev4-reusable-selection.yml","dev",{"round":"3"}),
                (True,"dev4-reusable-selection.yml","main",{"round":"2"}),
                (True,"dev4-reusable-selection.yml","main",{"round":"3","extra":"1"})]:
                with self.assertRaises(ValueError):
                    server.dispatch(workflow,ref,inputs,token="dummy",
                                    requester_approved=approved,api=self.api)
            self.assertEqual(self.calls,[])
            result=server.dispatch("dev4-reusable-selection.yml","main",{"round":"3"},
                                   token="dummy",requester_approved=True,api=self.api)
            self.assertTrue(result["accepted"])
            self.assertEqual([c[0] for c in self.calls],["GET","POST"])

    def test_duplicate_attempt_rejected(self):
        with patch.dict(server.ALLOWED,{"dev4-reusable-selection.yml":{"round":3}}):
            def duplicate(method,path,token,payload=None):
                self.calls.append(method)
                return {"total_count":1}
            with self.assertRaisesRegex(ValueError,"already exists"):
                server.dispatch("dev4-reusable-selection.yml","main",{"round":"3"},
                                token="dummy",requester_approved=True,api=duplicate)
            self.assertEqual(self.calls,["GET"])

    def test_run_status_read_only(self):
        def fake(method,path,token,payload=None):
            self.assertEqual(method,"GET")
            self.assertEqual(path,"/actions/runs/123")
            return {"id":123,"status":"completed","conclusion":"success","html_url":"https://github.com/example","run_attempt":1}
        self.assertEqual(server.status(123,token="dummy",api=fake)["conclusion"],"success")

if __name__=="__main__":unittest.main()
