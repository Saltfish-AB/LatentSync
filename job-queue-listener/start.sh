echo "Env: $ENV"
if [ "$ENV" = "development" ] ;
then
    echo "Starting in development mode"
    npm run watch
elif [ "$ENV" = "staging" ];
then
    npm run start
else
    npm run start
fi
