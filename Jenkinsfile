pipeline {
    agent {
        docker {
            image 'python:3.12-slim'
            args '-u root'
        }
    }

    stages {
        stage('Install') {
            steps {
                sh 'pip install --quiet -e ".[dev]"'
            }
        }

        stage('Lint') {
            steps {
                sh 'ruff check .'
            }
        }

        stage('Format') {
            steps {
                sh 'black --check .'
            }
        }

        stage('Type Check') {
            steps {
                sh 'mypy src/'
            }
        }

        stage('Test') {
            steps {
                sh 'pytest tests/ --junitxml=test-results.xml'
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'test-results.xml'
            cleanWs()
        }
    }
}
