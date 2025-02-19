import { BASE_URL } from "../Constants";

const useSearch = async () => {
  const data = await fetch(`${BASE_URL}/agents/`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  }).then((res) => res.json());

  return data;
};

export default useSearch;
