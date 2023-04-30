import os
from huggingface_hub import (
    # User management
    login,
    logout,
    whoami,

    # Repository creation and management
    create_repo,
    delete_repo,
    update_repo_visibility,

    # And some methods to retrieve/change information about the content
    list_models,
    list_datasets,
    list_metrics,
    list_repo_files,
    upload_file,
    delete_file,
)

usr = whoami()
print(usr['name'], usr['id'])

for fn in os.listdir("weights"):
    upload_file(
        path_or_fileobj=f"weights/{fn}",
        path_in_repo=fn,
        repo_id="maze/sd",
        repo_type="model"
    )

print(list_models(author='maze', limit=10))