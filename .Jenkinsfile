pipeline {
    agent any
    options {
        timeout(time:30, unit: 'MINUTES')
    }
    environment {
        PJ_NAME = 'fastNLP'
        POST_URL = 'https://open.feishu.cn/open-apis/bot/v2/hook/2f7122e3-3459-43d2-a9e4-ddd77bfc4282'
    }
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('Test Other'){
                    agent {
                        docker {
                            image 'fnlp:other'
                            args '-u root:root -v ${JENKINS_HOME}/html/docs:/docs -v ${JENKINS_HOME}/html/_ci:/ci'
                        }
                    }
                    steps {
                        sh 'pytest ./tests --durations=0 --html=other.html --self-contained-html -m "not (torch or paddle or paddledist or jittor or oneflow or oneflowdist or deepepeed or torchpaddle or torchjittor or torchoneflow)"'
                    }
                    post {
                        always {
                            sh 'html_path=/ci/${PJ_NAME}/report-${BUILD_NUMBER}-${GIT_BRANCH#*/}-${GIT_COMMIT} && mkdir -p ${html_path} && mv other.html ${html_path}'
                        }
                    }
                }
                stage('Test Torch-1.11') {
                    agent {
                        docker {
                            image 'fnlp:torch-1.11'
                            args '-u root:root -v ${JENKINS_HOME}/html/docs:/docs -v ${JENKINS_HOME}/html/_ci:/ci --gpus all --shm-size 1G'
                        }
                    }
                    steps {
                        sh 'pytest ./tests/ --durations=0 --html=torch-1.11.html --self-contained-html -m torch'
                    }
                    post {
                        always {
                            sh 'html_path=/ci/${PJ_NAME}/report-${BUILD_NUMBER}-${GIT_BRANCH#*/}-${GIT_COMMIT} && mkdir -p ${html_path} && mv torch-1.11.html ${html_path}'
                        }
                    }
                }
                stage('Test Torch-1.6') {
                    agent {
                        docker {
                            image 'fnlp:torch-1.6'
                            args '-u root:root -v ${JENKINS_HOME}/html/docs:/docs -v ${JENKINS_HOME}/html/_ci:/ci --gpus all'
                        }
                    }
                    steps {
                        sh 'pytest ./tests/ --durations=0 --html=torch-1.6.html --self-contained-html -m torch'
                    }
                    post {
                        always {
                            sh 'html_path=/ci/${PJ_NAME}/report-${BUILD_NUMBER}-${GIT_BRANCH#*/}-${GIT_COMMIT} && mkdir -p ${html_path} && mv torch-1.6.html ${html_path}'
                        }
                    }
                }
                stage('Test Paddle') {
                    agent {
                        docker {
                            image 'fnlp:paddle'
                            args '-u root:root -v ${JENKINS_HOME}/html/docs:/docs -v ${JENKINS_HOME}/html/_ci:/ci --gpus all'
                        }
                    }
                    steps {
                        sh 'pytest ./tests --durations=0 --html=paddle.html --self-contained-html -m paddle --co'
                        sh 'FASTNLP_BACKEND=paddle pytest ./tests --durations=0 --html=paddle_with_backend.html --self-contained-html -m paddle --co'
                        sh 'FASTNLP_BACKEND=paddle pytest ./tests/core/drivers/paddle_driver/test_dist_utils.py --durations=0 --html=paddle_dist_utils.html --self-contained-html --co'
                        sh 'FASTNLP_BACKEND=paddle pytest ./tests/core/drivers/paddle_driver/test_fleet.py --durations=0 --html=paddle_fleet.html --self-contained-html --co'
                        sh 'FASTNLP_BACKEND=paddle pytest ./tests/core/controllers/test_trainer_paddle.py --durations=0 --html=paddle_trainer.html --self-contained-html --co'
                    }
                    post {
                        always {
                            sh 'html_path=/ci/${PJ_NAME}/report-${BUILD_NUMBER}-${GIT_BRANCH#*/}-${GIT_COMMIT} && mkdir -p ${html_path} && mv paddle*.html ${html_path}'
                        }
                    }
                }
                // stage('Test Jittor') {
                //     agent {
                //         docker {
                //             image 'fnlp:jittor'
                //             args '-u root:root -v ${JENKINS_HOME}/html/docs:/docs -v ${JENKINS_HOME}/html/_ci:/ci --gpus all'
                //         }
                //     }
                //     steps {
                //         // sh 'pip install fitlog'
                //         // sh 'pytest ./tests --html=test_results.html --self-contained-html'
                //         sh 'pytest ./tests --durations=0 --html=jittor.html --self-contained-html -m jittor --co'
                //     }
                // }
            }
        }
    }
    post {
        failure {
            sh 'post 1'
        }
        success {
            sh 'post 0'
            // sh 'post github'
        }
    }
}