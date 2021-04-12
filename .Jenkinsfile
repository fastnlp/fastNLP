pipeline {
    agent {
        docker {
            image 'ubuntu_tester'
            args '-u root:root -v ${JENKINS_HOME}/html/docs:/docs -v ${JENKINS_HOME}/html/_ci:/ci'
        }
    }
    environment {
        TRAVIS = 1
        PJ_NAME = 'fastNLP'
        POST_URL = 'https://open.feishu.cn/open-apis/bot/v2/hook/14719364-818d-4f88-9057-7c9f0eaaf6ae'
    }
    stages {
        stage('Package Installation') {
            steps {
                sh 'python setup.py install'
            }
        }
        stage('Parallel Stages') {
            parallel {
                stage('Document Building') {
                    steps {
                        sh 'cd docs && make prod'
                        sh 'rm -rf /docs/${PJ_NAME}'
                        sh 'mv docs/build/html /docs/${PJ_NAME}'
                    }
                }
                stage('Package Testing') {
                    steps {
                        sh 'pip install fitlog'
                        sh 'pytest ./tests --html=test_results.html --self-contained-html'
                    }
                }
            }
        }
    }
    post {
        failure {
            sh 'post 1'
        }
        success {
            sh 'post 0'
            sh 'post github'
        }
    }

}