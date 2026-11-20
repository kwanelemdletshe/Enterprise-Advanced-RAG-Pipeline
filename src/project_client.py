import os

from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential


def create_project_client(credential: TokenCredential) -> AIProjectClient:
    connection_string = os.getenv("AZURE_PROJECT_CONNECTION_STRING")
    if connection_string:
        try:
            return AIProjectClient.from_connection_string(
                conn_str=connection_string,
                credential=credential,
            )
        except ValueError:
            pass

    endpoint = os.getenv("AZURE_PROJECT_ENDPOINT")
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    project_name = os.getenv("AZURE_PROJECT_NAME")

    if not endpoint:
        raise ValueError(
            "AZURE_PROJECT_ENDPOINT or AZURE_PROJECT_CONNECTION_STRING must be configured."
        )

    marker = "/api/projects/"
    if marker in endpoint:
        endpoint_project_name = endpoint.split(marker, 1)[1].strip("/")
    else:
        endpoint_project_name = project_name

    if not all((subscription_id, resource_group, project_name)):
        subscription_id = subscription_id or "placeholder"
        resource_group = resource_group or "placeholder"
        project_name = endpoint_project_name or "placeholder"

    return AIProjectClient(
        endpoint=endpoint,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        project_name=project_name,
        credential=credential,
    )
