# Demo evidence

- [Mac browser route](mac-browser-route.jpg) is an actual screenshot from the Mac browser over the LAN URL.
- [Trip 00000052](live_evaluation.json) records the actual HTTP run and scores. The operator independently confirmed that their Windows laptop opened the page and ran prediction over Wi-Fi.
- [Two more live trips](additional_trips.json) records the same model and decoder on 00000007 and 00000065. The latter produced a real GPS-gap break.
- [Uvicorn access log](server-requests.txt) records the observed endpoint requests with client addresses omitted.

The artifact contains aggregate metrics and a screenshot of the prediction. Windows screenshot and native Terminal screenshot were unavailable.
