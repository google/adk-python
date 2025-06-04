# OAuth Sample

## Introduction

This sample tests and demos the OAuth support in ADK via two tools:

* 1. list_calendar_events

  This is a customized tool that calls Google Calendar API to list calendar events.
  It pass in the client id and client secrete to ADK and then get back the access token from ADK.
  And then it uses the access token to call calendar api.

* 2. get_calendar_events

  This is an google calendar tool that calls Google Calendar API to get the details of a specific calendar.
  This tool is from the ADK built-in Google Calendar ToolSet.
  Everything is wrapped and the tool user just needs to pass in the client id and client secret.

## How to use

* 1. Follow https://developers.google.com/identity/protocols/oauth2#1.-obtain-oauth-2.0-credentials-from-the-dynamic_data.setvar.console_name. to get your client id and client secret.
  Be sure to choose "web" as your client type.

* 2. Configure your `.env` file to add two variables:

  * OAUTH_CLIENT_ID={your client id}
  * OAUTH_CLIENT_SECRET={your client secret}

  Note: don't create a separate `.env` file , instead put it to the same `.env` file that stores your Vertex AI or Dev ML credentials

* 3. Follow https://developers.google.com/identity/protocols/oauth2/web-server#creatingcred to add http://localhost/dev-ui/ to "Authorized redirect URIs".

  Note: localhost here is just a hostname that you use to access the dev ui, replace it with the actual hostname you use to access the dev ui.

* 4. For 1st run, allow popup for localhost in Chrome.

## Sample prompt

* `List all my today's meeting from 7am to 7pm.`
* `Get the details of the first event.`
