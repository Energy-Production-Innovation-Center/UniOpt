#!/usr/bin/env bash
# This script will automatically control the application version.

# Exit on error.
set -eo pipefail

api_url='https://dev01.unisim.cepetro.unicamp.br/api/v4/projects/22'
public_read_token='glpat-pZ7Yd7tJsA33x8rPvpGf'

get_development_version() {
    next_milestone=$(get_next_milestone)
    normalized_version=$(normalize_version "${next_milestone}")
    echo $(set_label "${normalized_version}" "dev")
}

get_stable_version() {
    last_release=$(get_last_release)
    commit_date=$(git log -1 --pretty=format:%cI)
    commit_date=$(date -d"${commit_date}" +%y%m%d%H%M%S)
    echo $(set_timestamp "${last_release}" "${commit_date}")
}

get_next_milestone() {
    url="$api_url/milestones?state=active"
    json=$(curl -sk --header "PRIVATE-TOKEN: $public_read_token" "$url")
    now=$(date +%s)
    next_milestone="0.0.1"

    while read i; do
        start=$(jq -r '.start_date' <<<$i)
        start=$(date -d $start +%s)
        if [[ $now > $start ]]; then
            due=$(jq -r '.due_date' <<<$i)
            due=$(date -d $due +%s)
            if [[ $now < $due ]]; then
                title=$(jq -r '.title' <<<$i)
                normalized_version=$(normalize_version "${title}")
                echo $normalized_version
                if is_stable_version "${normalized_version}"; then
                    next_milestone=$title
                    break
                fi
            fi
        fi
    done < <(jq -c '.[]' <<<$json)

    echo $next_milestone
}

get_last_release() {
    url="$api_url/releases"
    json=$(curl -sk --header "PRIVATE-TOKEN: $public_read_token" "$url")
    last_release="0.0.1"

    while read i; do
        tag_name=$(jq -r '.tag_name' <<<$i)
        normalized_version=$(normalize_version "${tag_name}")
        if is_stable_version "${normalized_version}"; then
            last_release=$tag_name
            break
        fi
    done < <(jq -c '.[]' <<<$json)

    echo $last_release
}

get_current_branch() {
    echo $(git rev-parse --abbrev-ref HEAD)
}

is_stable_version() {
    regex_pattern='^([0-9]+)\.([0-9]+)(\.([0-9]+))$'
    [[ "$1" =~ $regex_pattern ]]
}

normalize_version() {
    regex_pattern='^([0-9]+)\.([0-9]+)(\.([0-9]+))?'
    if [[ "$1" =~ $regex_pattern ]]; then
        year=${BASH_REMATCH[1]}
        month=${BASH_REMATCH[2]}
        minor=${BASH_REMATCH[4]}
        if [[ -z "${minor}" ]]; then
            minor=0
        fi
        echo "${year}.${month}.${minor}"
    else
        exit 1
    fi
}

set_timestamp() {
    regex_pattern='^([0-9]+)\.([0-9]+)\.([0-9]+)'
    if [[ "$1" =~ $regex_pattern ]]; then
        year=${BASH_REMATCH[1]}
        month=${BASH_REMATCH[2]}
        minor=${BASH_REMATCH[3]}
        timestamp=$2
        echo "${year}.${month}.${minor}.${timestamp}"
    else
        exit 1
    fi
}

set_label() {
    regex_pattern='^([0-9]+)\.([0-9]+)\.([0-9]+)'
    if [[ "$1" =~ $regex_pattern ]]; then
        year=${BASH_REMATCH[1]}
        month=${BASH_REMATCH[2]}
        minor=${BASH_REMATCH[3]}
        label=$2
        echo "${year}.${month}.${minor}-${label}"
    else
        exit 1
    fi
}


if [[ -z "${CI_COMMIT_TAG}" ]]; then
    if [[ ${CI_COMMIT_REF_NAME} == "main" || $(get_current_branch) == "main" ]]; then
        get_stable_version
    else
        get_development_version
    fi
else
    if is_stable_version "${CI_COMMIT_TAG}"; then
        echo "${CI_COMMIT_TAG}"
        exit 0
    else
        echo >&2 "Invalid version format from enviroment variable 'CI_COMMIT_TAG': ${CI_COMMIT_TAG}"
        exit 1
    fi
fi
