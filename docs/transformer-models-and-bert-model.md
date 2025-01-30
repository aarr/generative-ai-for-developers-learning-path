# [Transformer Models and BERT Model](https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/text_models/solutions/classify_text_with_bert.ipynb)

* GCPでNodebook作成
  * CloudShellから実行

    [main.tf](https://github.com/terraform-google-modules/terraform-docs-samples/blob/main/vertex_ai/user_managed_notebooks_instance/main.tf)にprojectを追加し、実行

    ![project](./img/gcp-project.png)  

    ```bash
    # Terraformファイル作成
    mkdir terraform; vi terraform/main.tf
    # Terraform初期化
    terraform init
    # 事前確認
    terraform plan
    # 正常に実行できたら反映
    terraform apply
    ```

  * main.tf

    ```tf
    # [START aiplatform_create_user_managed_notebooks_instance_sample]
    resource "google_notebooks_instance" "basic_instance" {
      project      = "設定"
      name         = "notebooks-instance-basic"
      location     = "us-central1-a"
      machine_type = "e2-medium"

      vm_image {
        project      = "deeplearning-platform-release"
        image_family = "tf-ent-2-9-cu113-notebooks"
      }
    }
    # [END aiplatform_create_user_managed_notebooks_instance_sample]
    ```
